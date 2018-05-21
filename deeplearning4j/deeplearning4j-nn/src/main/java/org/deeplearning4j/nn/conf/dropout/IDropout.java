package org.deeplearning4j.nn.conf.dropout;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.shade.jackson.annotation.JsonTypeInfo;

import java.io.Serializable;

/**
 * IDropout instances operate on an activations array, modifying or dropping values at training time only.
 * IDropout instances are not applied at test time.
 *
 * @author Alex Black
 */
@JsonTypeInfo(use = JsonTypeInfo.Id.CLASS, include = JsonTypeInfo.As.PROPERTY, property = "@class")
public interface IDropout extends Serializable, Cloneable {

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

    IDropout clone();
}
