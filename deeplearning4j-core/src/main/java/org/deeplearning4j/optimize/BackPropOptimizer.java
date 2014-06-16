package org.deeplearning4j.optimize;

import cc.mallet.optimize.Optimizable;
import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.berkeley.Triple;
import org.deeplearning4j.nn.BaseMultiLayerNetwork;
import org.deeplearning4j.nn.NeuralNetwork;
import org.deeplearning4j.nn.OutputLayer;
import org.jblas.DoubleMatrix;
import org.jblas.ranges.RangeUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.Serializable;
import java.util.List;

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

    public BackPropOptimizer(BaseMultiLayerNetwork network,double lr,int epochs) {
        this.network = network;
        this.lr = lr;
        this.epochs = epochs;
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
                    network.backPropStep(lr);
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
                    network.backPropStep(lr);
                /* Trains logistic regression post weight updates */
                    network.getOutputLayer().trainTillConvergence(lr, epochs);

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
                g.setTolerance(1e-3);
                g.setTrainingEvaluator(eval);
                g.setMaxIterations(numEpochs);
                g.optimize(numEpochs);

            }

            else {
                VectorizedDeepLearningGradientAscent g = new VectorizedDeepLearningGradientAscent(this);
                g.setTolerance(1e-3);
                g.setTrainingEvaluator(eval);
                g.optimize(numEpochs);

            }

        }

    }

    @Override
    public void getValueGradient(double[] buffer) {
        System.arraycopy(network.params().data,0,buffer,0,buffer.length);
    }

    @Override
    public double getValue() {
        return - (network.score());
    }

    @Override
    public int getNumParameters() {
        if(length < 0)
            length = getParameters().length;
        return length;
    }

    @Override
    public void getParameters(double[] buffer) {
        System.arraycopy(getParameters().data,0,buffer,0,buffer.length);
    }

    @Override
    public double getParameter(int index) {
        return 0;
    }

    @Override
    public void setParameters(double[] params) {
        setParameters(new DoubleMatrix(params).reshape(1,params.length));
    }

    @Override
    public void setParameter(int index, double value) {

    }

    @Override
    public DoubleMatrix getParameters() {
        return network.params();
    }

    @Override
    public void setParameters(DoubleMatrix params) {
        for(int i = 0; i < network.getLayers().length; i++) {
            ParamRange range = startIndexForLayer(i);
            DoubleMatrix w = params.get(RangeUtils.all(),RangeUtils.interval(range.getwStart(),range.getwEnd()));
            DoubleMatrix bias = params.get(RangeUtils.all(),RangeUtils.interval(range.getBiasStart(),range.getBiasEnd()));
            int rows = network.getLayers()[i].getW().rows,columns = network.getLayers()[i].getW().columns;
            network.getLayers()[i].setW(w.reshape(rows,columns));
            network.getLayers()[i].sethBias(bias.reshape(network.getLayers()[i].gethBias().rows,network.getLayers()[i].gethBias().columns));
        }


        ParamRange range = startIndexForLayer(network.getLayers().length);
        DoubleMatrix w = params.get(RangeUtils.all(),RangeUtils.interval(range.getwStart(),range.getwEnd()));
        DoubleMatrix bias = params.get(RangeUtils.all(),RangeUtils.interval(range.getBiasStart(),range.getBiasEnd()));
        int rows = network.getOutputLayer().getW().rows,columns = network.getOutputLayer().getW().columns;
        network.getOutputLayer().setW(w.reshape(rows, columns));
        network.getOutputLayer().setB(bias.reshape(network.getOutputLayer().getB().rows, network.getOutputLayer().getB().columns));

        network.getOutputLayer().trainTillConvergence(lr,epochs);


    }

    @Override
    public DoubleMatrix getValueGradient(int iteration) {

        return network.getBackPropGradient();
    }


    public ParamRange startIndexForLayer(int layer) {
        int start = 0;
        for(int i = 0; i < layer; i++) {
            start += network.getLayers()[i].getW().length;
            start += network.getLayers()[i].gethBias().length;
        }
        if(layer < network.getLayers().length) {
            int wEnd = start + network.getLayers()[layer].getW().length;
            return new ParamRange(start,wEnd,wEnd,wEnd + network.getLayers()[layer].gethBias().length);

        }

        else {
            int wEnd = start + network.getOutputLayer().getW().length;
            return new ParamRange(start,wEnd,wEnd,wEnd + network.getOutputLayer().getB().length);

        }


    }

    public static class ParamRange implements  Serializable {
        private int wStart,wEnd,biasStart,biasEnd;

        private ParamRange(int wStart, int wEnd, int biasStart, int biasEnd) {
            this.wStart = wStart;
            this.wEnd = wEnd;
            this.biasStart = biasStart;
            this.biasEnd = biasEnd;
        }

        public int getwStart() {
            return wStart;
        }

        public void setwStart(int wStart) {
            this.wStart = wStart;
        }

        public int getwEnd() {
            return wEnd;
        }

        public void setwEnd(int wEnd) {
            this.wEnd = wEnd;
        }

        public int getBiasStart() {
            return biasStart;
        }

        public void setBiasStart(int biasStart) {
            this.biasStart = biasStart;
        }

        public int getBiasEnd() {
            return biasEnd;
        }

        public void setBiasEnd(int biasEnd) {
            this.biasEnd = biasEnd;
        }
    }


}
