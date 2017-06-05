package org.deeplearning4j.parallelism.trainer;

import org.deeplearning4j.optimize.listeners.SharedGradient;

/**
 * @author raver119@gmail.com
 */
public interface CommunicativeTrainer extends Trainer {

    void enqueueGradient(SharedGradient gradient);
}
