package org.deeplearning4j.optimize.api;

import java.io.Serializable;

/**
 *
 * Training evaluator, used for determining early stop
 *
 * @author Adam Gibson
 */
public interface TrainingEvaluator extends Serializable {

    /**
     * Whether to terminate or  not
     * @param epoch the current epoch
     * @return whether to terminate or not
     * on the given epoch
     */
    boolean shouldStop(int epoch);

    public double improvementThreshold();


    double patience();


    /**
     * Amount patience should be increased when a new best threshold is hit
     * @return
     */
    double patienceIncrease();


    /**
     * The best validation loss so far
     * @return the best validation loss so far
     */
    public double bestLoss();

    /**
     * The number of epochs to test on
     * @return the number of epochs to test on
     */
    public int validationEpochs();


    public int miniBatchSize();

}
