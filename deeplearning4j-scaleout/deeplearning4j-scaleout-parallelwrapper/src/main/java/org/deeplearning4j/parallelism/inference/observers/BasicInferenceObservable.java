package org.deeplearning4j.parallelism.inference.observers;

import lombok.Data;
import lombok.Getter;
import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.parallelism.inference.InferenceObservable;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Observable;

/**
 * This class holds reference input, and implements basic use case: SEQUENTIAL inference
 */
@Slf4j
public class BasicInferenceObservable extends Observable implements InferenceObservable {
    @Getter
    private INDArray[] input;
    @Getter
    private long id;
    @Getter
    private INDArray[] output;


    public BasicInferenceObservable(INDArray... inputs) {
        super();
        this.input = inputs;
    }

    @Override
    public void setInput(INDArray... input) {
        this.input = input;
    }

    public void setOutput(INDArray... output) {
        this.output = output;
        this.setChanged();
        notifyObservers();
    }
}
