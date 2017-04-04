package org.deeplearning4j.parallelism.inference.observers;

import org.deeplearning4j.parallelism.inference.InferenceObservable;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.exception.ND4JIllegalStateException;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * This class holds reference input, and implements second use case: BATCHED inference
 *
 * @author raver119@gmail.com
 */
public class BatchedInferenceObservable extends BasicInferenceObservable implements InferenceObservable {
    private List<INDArray[]> inputs = new ArrayList<>();
    private List<INDArray[]> outputs = new ArrayList<>();
    private AtomicInteger counter = new AtomicInteger(0);
    private ThreadLocal<Integer> position = new ThreadLocal<>();

    private final Object locker = new Object();

    public BatchedInferenceObservable() {

    }

    @Override
    public void setInput(INDArray... input) {
        synchronized (locker) {
            inputs.add(input);
            position.set(counter.getAndIncrement());
        }
    }

    @Override
    public INDArray[] getInput() {
        // this method should pile individual examples into single batch
        INDArray[] result = new INDArray[inputs.get(0).length];
        for(int i = 0; i < result.length; i++) {
            List<INDArray> examples = new ArrayList<>();
            for (int e = 0; e < inputs.size(); e++) {
                examples.add(inputs.get(e)[i]);
            }
            result[i] = Nd4j.pile(examples);
        }

        return result;
    }

    @Override
    public void setOutput(INDArray... output) {
        //this method should split batched output INDArray[] into multiple separate INDArrays
        // pre-create outputs
        for(int i = 0; i < counter.get(); i++) {
            outputs.add(new INDArray[output.length]);
        }

        // pull back results for individual examples
        int cnt = 0;
        for(INDArray array: output) {
            int[] dimensions = new int[array.rank() - 1];
            for (int i = 1; i < array.rank(); i++) {
                dimensions[i-1] = i;
            }

            INDArray[] split = Nd4j.tear(array, dimensions);
            if (split.length != counter.get())
                throw new ND4JIllegalStateException("Number of splits doesn't match number of queries");

            for(int e = 0; e < counter.get(); e++) {
                outputs.get(e)[cnt] = split[e];
            }
            cnt++;
        }

        notifyObservers();
    }

    /**
     * PLEASE NOTE: This method is for tests only
     *
     * @return
     */
    protected List<INDArray[]> getOutputs(){
        return outputs;
    }

    protected void setCounter(int value) {
        counter.set(value);
    }

    @Override
    public INDArray[] getOutput() {
        // basically we should take care of splits here: each client should get its own part of output, wrt order number

        return outputs.get(position.get());
    }
}
