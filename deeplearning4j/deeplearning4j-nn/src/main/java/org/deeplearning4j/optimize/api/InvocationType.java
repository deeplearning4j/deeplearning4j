package org.deeplearning4j.optimize.api;

/**
 * This enum holds options for TrainingListener invocation scheme
 *
 * @author raver119@gmail.com
 */
public enum InvocationType {
    /**
     * Iterator will be called on start of epoch.
     * PLEASE NOTE: This option makes sense only for pretrained models.
     */
    EPOCH_START,

    /**
     * Iterator will be called on end of epoch
     */
    EPOCH_END,

    /**
     * Iterator will be called on iteration end
     */
    ITERATION_END,
}
