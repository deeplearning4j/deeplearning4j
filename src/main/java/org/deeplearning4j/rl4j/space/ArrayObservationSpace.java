package org.deeplearning4j.rl4j.space;

import lombok.Value;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 * @param <O> the type of Observation
 * @author rubenfiszel (ruben.fiszel@epfl.ch) on 7/8/16.
 *         <p>
 *         An array observation space enables you to create an Observation Space of custom dimension
 */

@Value
public class ArrayObservationSpace<O> implements ObservationSpace<O> {

    String name;
    int[] shape;
    INDArray low;
    INDArray high;

    public ArrayObservationSpace(int[] shape) {
        name = "Custom";
        this.shape = shape;
        low = Nd4j.create(1);
        high = Nd4j.create(1);
    }

}
