package org.deeplearning4j.rl4j.learning.sync;

import lombok.Value;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 * @author rubenfiszel (ruben.fiszel@epfl.ch) 7/12/16.
 */
@Value
public class Transition<A> {

    INDArray[] observation;
    A action;
    double reward;
    boolean isTerminal;
    INDArray[] nextObservation;

    public static INDArray concat(INDArray[] history){
        INDArray[] decompressed = new INDArray[history.length];
        for (int i = 0; i < history.length; i++) {
            decompressed[i] = history[i].dup();
        }
        INDArray arr = Nd4j.concat(0, decompressed);
        if (arr.shape().length > 2)
            arr.muli(1/256f);
        return arr;
    }

}
