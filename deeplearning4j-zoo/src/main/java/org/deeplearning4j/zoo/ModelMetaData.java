package org.deeplearning4j.zoo;

import lombok.AllArgsConstructor;
import lombok.Getter;

/**
 * Metadata describing a model, including input shapes. This is helpful for instantiating
 * the model programmatically and ensuring appropriate inputs are used.
 */
@Getter
@AllArgsConstructor
public class ModelMetaData {
    private int[][] inputShape;
    private int numOutputs;
    private ZooType zooType;

    /**
     * If number of inputs are greater than 1, this states that the
     * implementation should use MultiDataSet.
     * @return
     */
    public boolean useMDS() {
        return inputShape.length > 1 ? true : false;
    }
}
