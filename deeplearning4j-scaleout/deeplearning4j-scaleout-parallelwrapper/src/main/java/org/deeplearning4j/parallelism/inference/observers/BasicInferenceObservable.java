package org.deeplearning4j.parallelism.inference.observers;

import com.google.common.base.Preconditions;
import lombok.Getter;
import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.parallelism.inference.InferenceObservable;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Collections;
import java.util.List;
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
    public void addInput(@NonNull INDArray... input) {
        this.input = input;
    }

    @Override
    public void setOutputBatches(@NonNull List<INDArray[]> output) {
        Preconditions.checkArgument(output.size() == 1, "Expected size 1 output: got size " + output.size());
        this.output = output.get(0);
        this.setChanged();
        notifyObservers();
    }

    @Override
    public List<INDArray[]> getInputBatches(){
        return Collections.singletonList(input);
    }
}
