package org.deeplearning4j.optimize.optimizers;

import org.deeplearning4j.linalg.api.ndarray.INDArray;
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
 * Optimizes via back prop gradients
 * @author Adam Gibson
 */
public class BackPropOptimizer implements Serializable,OptimizableByGradientValueMatrix {

    private BaseMultiLayerNetwork network;
    private int length = -1;
    private float lr  = 1e-1f;
    private int epochs = 1000;
    private static Logger log = LoggerFactory.getLogger(BackPropOptimizer.class);
    private int currentIteration = -1;

    public BackPropOptimizer(BaseMultiLayerNetwork network,float lr,int epochs) {
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
                    if(i % network.getResetAdaGradIterations() == 0)
                        network.getOutputLayer().getAdaGrad().historicalGradient = null;
                    network.backPropStep();
                    log.info("Iteration " + i + " error " + network.score());

                }
            }

            else {


                boolean train = true;
                int count = 0;
                float changeTolerance = 1e-5f;
                int backPropIterations = 0;
                while(train) {
                    if(backPropIterations >= epochs) {
                        log.info("Backprop number of iterations max hit; converging");
                        break;

                    }
                    count++;
                    network.backPropStep();

                /* Trains logistic regression post weight updates */

                    Float entropy = network.score();
                    if(lastEntropy == null || entropy < lastEntropy) {
                        float diff = Math.abs(entropy - lastEntropy);
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
    public float getValue() {
        return - (network.score());
    }

    @Override
    public int getNumParameters() {
        if(length < 0)
            length = getParameters().length();
        return length;
    }




    @Override
    public void setParameter(int index, float value) {

    }

    @Override
    public INDArray getParameters() {
        return network.params();
    }

    @Override
    public float getParameter(int index) {
        return 0;
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
