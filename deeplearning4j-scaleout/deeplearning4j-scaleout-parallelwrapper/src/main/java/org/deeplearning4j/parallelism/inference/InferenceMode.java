package org.deeplearning4j.parallelism.inference;

/**
 * @author raver119@gmail.com
 */
public enum InferenceMode {
    SEQUENTIAL, // input will be passed into the model as is
    BATCHED, // input will be included into the batch
}
