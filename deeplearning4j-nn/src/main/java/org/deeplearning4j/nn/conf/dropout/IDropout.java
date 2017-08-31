package org.deeplearning4j.nn.conf.dropout;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.shade.jackson.annotation.JsonTypeInfo;

@JsonTypeInfo(use = JsonTypeInfo.Id.CLASS, include = JsonTypeInfo.As.PROPERTY, property = "@class")
public interface IDropout {

    /**
     *
     * @param inputActivations Input activations array
     * @param iteration        Current iteration number
     * @param epoch            Current epoch number
     * @param inPlace          If true: modify the input activations in-place. False: Copy the input activations and
     *                         apply dropout on the copy instead
     * @return
     */
    INDArray applyDropout(INDArray inputActivations, int iteration, int epoch, boolean inPlace);


}
