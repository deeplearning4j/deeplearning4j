package org.deeplearning4j.optimize;

import cc.mallet.optimize.Optimizable;
import org.deeplearning4j.linalg.api.ndarray.INDArray;
import org.deeplearning4j.linalg.factory.NDArrays;
import org.deeplearning4j.nn.BaseMultiLayerNetwork;

import org.deeplearning4j.nn.NeuralNetwork;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.Serializable;

/**
 * Optimizes via back prop gradients
 * @author Adam Gibson
 */
public class BackPropOptimizer implements Optimizable.ByGradientValue,Serializable,OptimizableByGradientValueMatrix {

    private BaseMultiLayerNetwork network;
    private int length = -1;
    private double lr  = 1e-1;
    private int epochs = 1000;
    private static Logger log = LoggerFactory.getLogger(BackPropOptimizer.class);
    private int currentIteration = -1;

    public BackPropOptimizer(BaseMultiLayerNetwork network,double lr,int epochs) {
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

            Double lastEntropy =  network.score();
            //store a copy of the network for when binary cross entropy gets
            //worse after an iteration
            BaseMultiLayerNetwork revert = network.clone();
            //sgd style; only train a certain number of epochs
            if(network.isForceNumEpochs()) {
                for(int i = 0; i < epochs; i++) {
                    if(i % network.getResetAdaGradIterations() == 0)
                        network.getOutputLayer().getAdaGrad().historicalGradient = null;
                    network.backPropStep();
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
                    network.backPropStep();

                /* Trains logistic regression post weight updates */

                    Double entropy = network.score();
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

                    else if(entropy >= lastEntropy) {
                        train = false;
                        network.update(revert);
                        log.info("Reverting to best score " + lastEntropy);
                    }

                    backPropIterations++;
                }


            }
        }

        else {

            NeuralNetwork.OptimizationAlgorithm optimizationAlgorithm = network.getOptimizationAlgorithm();
            if(optimizationAlgorithm == NeuralNetwork.OptimizationAlgorithm.CONJUGATE_GRADIENT) {
                VectorizedNonZeroStoppingConjugateGradient g = new VectorizedNonZeroStoppingConjugateGradient(this);
                g.setTrainingEvaluator(eval);
                g.setMaxIterations(numEpochs);
                g.optimize(numEpochs);

            }

            else if(optimizationAlgorithm == NeuralNetwork.OptimizationAlgorithm.HESSIAN_FREE) {
                StochasticHessianFree s = new StochasticHessianFree(this,network);
                s.setTrainingEvaluator(eval);
                s.setMaxIterations(numEpochs);
                s.optimize(numEpochs);

            }



            else {
                VectorizedDeepLearningGradientAscent g = new VectorizedDeepLearningGradientAscent(this);
                g.setTrainingEvaluator(eval);
                g.optimize(numEpochs);

            }

        }

        network.getOutputLayer().trainTillConvergence(lr,numEpochs,eval);


    }

    @Override
    public void getValueGradient(double[] buffer) {
       throw new UnsupportedOperationException();
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
    public void getParameters(double[] buffer) {
        System.arraycopy(getParameters().data(),0,buffer,0,buffer.length);
    }

    @Override
    public double getParameter(int index) {
        return 0;
    }

    @Override
    public void setParameters(double[] params) {
        setParameters(NDArrays.create(params).reshape(1,params.length));
    }

    @Override
    public void setParameter(int index, double value) {

    }

    @Override
    public INDArray getParameters() {
        return network.params();
    }

    @Override
    public void setParameters(INDArray params) {
        network.setParameters(params);
        network.getOutputLayer().trainTillConvergence(lr,epochs);


    }

    @Override
    public INDArray getValueGradient(int iteration) {
        return network.getBackPropGradient2().getFirst();
    }




}
