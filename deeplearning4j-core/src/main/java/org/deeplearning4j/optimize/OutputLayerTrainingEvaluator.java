package org.deeplearning4j.optimize;

import org.deeplearning4j.datasets.DataSet;
import org.deeplearning4j.nn.BaseMultiLayerNetwork;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * An output layer training evaluator uses a multi layer networks output and score functions to determine if
 * training iterations should continue
 * @author Adam Gibson
 */
public class OutputLayerTrainingEvaluator implements TrainingEvaluator {

    private BaseMultiLayerNetwork network;
    private double patience;
    private double patienceIncrease;
    private double bestLoss;
    private int validationEpochs;
    private int miniBatchSize;
    private DataSet testSet;
    private double improvementThreshold;
    private static Logger log = LoggerFactory.getLogger(OutputLayerTrainingEvaluator.class);


    public OutputLayerTrainingEvaluator(BaseMultiLayerNetwork network, double patience, double patienceIncrease, double bestLoss, int validationEpochs, int miniBatchSize, DataSet testSet, double improvementThreshold) {
        this.network = network;
        this.patience = 4 * miniBatchSize;
        this.patienceIncrease = patienceIncrease;
        this.bestLoss = bestLoss;
        this.validationEpochs = validationEpochs;
        this.miniBatchSize = miniBatchSize;
        this.testSet = testSet;
        this.improvementThreshold = improvementThreshold;
    }

    /**
     * Whether to terminate or  not
     *
     * @param epoch the current epoch
     * @return whether to terminate or not
     * on the given epoch
     */
    @Override
    public boolean shouldStop(int epoch) {
        if(!(epoch % validationEpochs == 0) || epoch < 2)
            return false;
        double score = network.score();
        if(score < bestLoss) {
            if(score < bestLoss * improvementThreshold) {
                bestLoss = score;
                patience = Math.max(patience,epoch * patienceIncrease);


            }

        }
        boolean ret =  patience < epoch;
        if(ret) {
            log.info("Returning early on finetune");
        }

        return ret;
    }

    /**
     * Amount patience should be increased when a new best threshold is hit
     *
     * @return
     */
    @Override
    public double patienceIncrease() {
        return patienceIncrease;
    }

    @Override
    public double improvementThreshold() {
        return improvementThreshold;
    }

    @Override
    public double patience() {
        return patience;
    }


    /**
     * The best validation loss so far
     *
     * @return the best validation loss so far
     */
    @Override
    public double bestLoss() {
        return bestLoss;
    }

    /**
     * The number of epochs to test on
     *
     * @return the number of epochs to test on
     */
    @Override
    public int validationEpochs() {
        return validationEpochs;
    }

    @Override
    public int miniBatchSize() {
        return miniBatchSize;
    }


    public static class Builder {
        private BaseMultiLayerNetwork network;
        private double patience;
        private double patienceIncrease;
        private double bestLoss;
        private int validationEpochs;
        private int miniBatchSize;
        private DataSet testSet;
        private double improvementThreshold;


        public Builder withNetwork(BaseMultiLayerNetwork network) {
            this.network = network;
            return this;
        }

        public Builder patience(double patience) {
            this.patience = patience;
            return this;
        }


        public Builder patienceIncrease(double patienceIncrease) {
            this.patienceIncrease = patienceIncrease;
            return this;
        }

        public Builder bestLoss(double bestLoss) {
            this.bestLoss = bestLoss;
            return this;
        }

        public Builder validationEpochs(int validationEpochs) {
            this.validationEpochs = validationEpochs;
            return this;
        }

        public Builder testSet(DataSet testSet) {
            this.testSet = testSet;
            return this;
        }

        public Builder miniBatchSize(int miniBatchSize) {
            this.miniBatchSize = miniBatchSize;
            return this;
        }

        public Builder improvementThreshold(double improvementThreshold) {
            this.improvementThreshold = improvementThreshold;
            return this;
        }

        public OutputLayerTrainingEvaluator build() {
            return new  OutputLayerTrainingEvaluator(network,patience,patienceIncrease,bestLoss,validationEpochs, miniBatchSize, testSet,improvementThreshold);
        }









    }

}
