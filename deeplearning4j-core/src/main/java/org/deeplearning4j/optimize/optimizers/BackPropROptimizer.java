package org.deeplearning4j.optimize.optimizers;

import org.deeplearning4j.linalg.api.ndarray.INDArray;
import org.deeplearning4j.linalg.factory.NDArrays;
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
 * Optimizes via back prop gradients with
 * the r operator, used in hessian free operators
 * @author Adam Gibson
 */
public class BackPropROptimizer implements Serializable,OptimizableByGradientValueMatrix {

    private BaseMultiLayerNetwork network;
    private int length = -1;
    private double lr  = 1e-1;
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

    public void optimize(TrainingEvaluator eval,int numEpochs,boolean lineSearch) {
        if(!lineSearch) {
            log.info("BEGIN BACKPROP WITH SCORE OF " + network.score());

            Float lastEntropy =  network.score();
            //store a copy of the network for when binary cross entropy gets
            //worse after an iteration
            BaseMultiLayerNetwork revert = network.clone();
            //sgd style; only iterate a certain number of epochs
            if(network.isForceNumEpochs()) {
                for(int i = 0; i < epochs; i++) {
                    if(i % network.getDefaultConfiguration().getResetAdaGradIterations() == 0)
                        network.getOutputLayer().getAdaGrad().historicalGradient = null;
                    network.backPropStepR(null);
                    log.info("Iteration " + i + " error " + network.score());

                }
            }

            else {


                boolean train = true;
                int count = 0;
                double changeTolerance = 1e-5;
                int backPropIterations = 0;
                while(train) {
                    if(backPropIterations >= epochs) {
                        log.info("Backprop number of iterations max hit; converging");
                        break;

                    }
                    count++;
                /* Trains logistic regression post weight updates */

                    Float entropy = network.score();
                    if(lastEntropy == null || entropy < lastEntropy) {
                        double diff = Math.abs(entropy - lastEntropy);
                        if(diff < changeTolerance) {
                            log.info("Not enough of a change on back prop...breaking");
                            break;
                        }
                        else
                            lastEntropy = entropy;
                        log.info("New score " + lastEntropy);
                        revert = network.clone();
                    }

                    else if(count >= epochs) {
                        log.info("Hit max number of epochs...breaking");
                        train = false;
                    }

                    else if(entropy >= lastEntropy || Float.isNaN(entropy) || Float.isInfinite(entropy)) {
                        train = false;
                        network.update(revert);
                        log.info("Reverting to best score " + lastEntropy);
                    }

                    backPropIterations++;
                }


            }
        }

        else {

            NeuralNetwork.OptimizationAlgorithm optimizationAlgorithm = network.getDefaultConfiguration().getOptimizationAlgo();
            if(optimizationAlgorithm == NeuralNetwork.OptimizationAlgorithm.CONJUGATE_GRADIENT) {
                VectorizedNonZeroStoppingConjugateGradient g = new VectorizedNonZeroStoppingConjugateGradient(this);
                g.setTrainingEvaluator(eval);
                g.setMaxIterations(numEpochs);
                g.optimize(numEpochs);

            }

            else if(optimizationAlgorithm == NeuralNetwork.OptimizationAlgorithm.HESSIAN_FREE) {
                h = new StochasticHessianFree(this,network);
                h.setTrainingEvaluator(eval);
                h.optimize(numEpochs);
            }

            else {
                VectorizedDeepLearningGradientAscent g = new VectorizedDeepLearningGradientAscent(this);
                g.setTrainingEvaluator(eval);
                g.optimize(numEpochs);

            }

        }

    }


    public void getValueGradient(double[] buffer) {
        System.arraycopy(network.getBackPropRGradient(network.params()).data(),0,buffer,0,buffer.length);
    }

    @Override
    public float getValue() {
        return - (network.score());
    }

    @Override
    public int getNumParameters() {
        if(length < 0)
            length = getParameters().length();
        return length;
    }


    public void getParameters(float[] buffer) {
        System.arraycopy(getParameters().data(),0,buffer,0,buffer.length);
    }

    @Override
    public float getParameter(int index) {
        return 0;
    }


    public void setParameters(float[] params) {
        setParameters(NDArrays.create(params));
    }


    public void setParameter(int index, double value) {

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
    public void setParameter(int index, float value) {

    }

    @Override
    public INDArray getValueGradient(int iteration) {
        return network.getBackPropRGradient(network.params());
    }



}
