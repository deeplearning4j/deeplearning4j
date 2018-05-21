package org.deeplearning4j.parallelism.inference.observers;

import com.google.common.base.Preconditions;
import lombok.Getter;
import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.parallelism.inference.InferenceObservable;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.primitives.Pair;

import java.util.Collections;
import java.util.List;
import java.util.Observable;

/**
 * This class holds reference input, and implements basic use case: SEQUENTIAL inference
 */
@Slf4j
public class BasicInferenceObservable extends Observable implements InferenceObservable {
    private INDArray[] input;
    private INDArray[] inputMasks;
    @Getter
    private long id;
    private INDArray[] output;
    protected Exception exception;


    public BasicInferenceObservable(INDArray... inputs) {
        this(inputs, null);
    }

    public BasicInferenceObservable(INDArray[] inputs, INDArray[] inputMasks){
        super();
        this.input = inputs;
        this.inputMasks = inputMasks;
    }

    @Override
    public void addInput(@NonNull INDArray... input){
        addInput(input, null);
    }

    @Override
    public void addInput(@NonNull INDArray[] input, INDArray[] inputMasks) {
        this.input = input;
        this.inputMasks = inputMasks;
    }

    @Override
    public void setOutputBatches(@NonNull List<INDArray[]> output) {
        Preconditions.checkArgument(output.size() == 1, "Expected size 1 output: got size " + output.size());
        this.output = output.get(0);
        this.setChanged();
        notifyObservers();
    }

    @Override
    public List<Pair<INDArray[],INDArray[]>> getInputBatches(){
        return Collections.singletonList(new Pair<>(input, inputMasks));
    }

    @Override
    public void setOutputException(Exception exception){
        this.exception = exception;
        this.setChanged();
        notifyObservers();
    }

    @Override
    public INDArray[] getOutput(){
        checkOutputException();
        return output;
    }

    protected void checkOutputException(){
        if(exception != null){
            if(exception instanceof RuntimeException){
                throw (RuntimeException)exception;
            } else {
                throw new RuntimeException("Exception encountered while getting output: " + exception.getMessage(), exception);
            }
        }
    }
}
