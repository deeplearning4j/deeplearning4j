package org.deeplearning4j.parallelism.inference.observers;

import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.parallelism.inference.InferenceObservable;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.exception.ND4JIllegalStateException;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.locks.ReentrantReadWriteLock;

/**
 * This class holds reference input, and implements second use case: BATCHED inference
 *
 * @author raver119@gmail.com
 */
@Slf4j
public class BatchedInferenceObservable extends BasicInferenceObservable implements InferenceObservable {
    private List<INDArray[]> inputs = new ArrayList<>();
    private List<INDArray[]> outputs = new ArrayList<>();
    private AtomicInteger counter = new AtomicInteger(0);
    private ThreadLocal<Integer> position = new ThreadLocal<>();

    private final Object locker = new Object();

    private ReentrantReadWriteLock realLocker = new ReentrantReadWriteLock();
    private AtomicBoolean isLocked = new AtomicBoolean(false);
    private AtomicBoolean isReadLocked = new AtomicBoolean(false);

    public BatchedInferenceObservable() {

    }

    @Override
    public void setInput(INDArray... input) {
        synchronized (locker) {
            inputs.add(input);
            position.set(counter.getAndIncrement());

            if (isReadLocked.get())
                realLocker.readLock().unlock();
        }
    }

    @Override
    public INDArray[] getInput() {
        realLocker.writeLock().lock();
        isLocked.set(true);

        // this method should pile individual examples into single batch
        if (counter.get() > 1) {
            INDArray[] result = new INDArray[inputs.get(0).length];

            //First: determine which we can actually batch...
            int lastPossible = 0;
            for( int i=1; i<inputs.size(); i++ ){
                if(canBatch(inputs.get(0), inputs.get(i))){
                    lastPossible = i;
                } else {
                    break;
                }
            }

            for (int i = 0; i < result.length; i++) {
                List<INDArray> examples = new ArrayList<>();
                for (int e = 0; e <= lastPossible; e++) {
                    examples.add(inputs.get(e)[i]);
                }
                result[i] = Nd4j.pile(examples);
            }

            realLocker.writeLock().unlock();
            return result;
        } else {
            realLocker.writeLock().unlock();
            return inputs.get(0);
        }
    }

    private static boolean canBatch(INDArray[] first, INDArray[] candidate){
        //For now: let's simply require that the inputs have the same shape
        //In the future: we'll intelligently handle the RNN variable length case
        for(int i=0; i<first.length; i++ ){
            if(!Arrays.equals(first[i].shape(), candidate[i].shape())){
                return false;
            }
        }
        return true;
    }

    @Override
    public void setOutput(INDArray... output) {
        //this method should split batched output INDArray[] into multiple separate INDArrays
        // pre-create outputs
        int exampleCount = output[0].size(0);
        if(exampleCount == 1){
            outputs.add(output);
        } else {
            for (int i = 0; i < exampleCount; i++) {
                outputs.add(new INDArray[output.length]);
            }

            // pull back results for individual examples
            int cnt = 0;
            for (INDArray array : output) {
                int[] dimensions = new int[array.rank() - 1];
                for (int i = 1; i < array.rank(); i++) {
                    dimensions[i - 1] = i;
                }

                INDArray[] split = Nd4j.tear(array, dimensions);

                for (int e = 0; e < exampleCount; e++) {
                    outputs.get(e)[cnt] = split[e];
                }
                cnt++;
            }
        }

        this.setChanged();
        notifyObservers();
    }

    /**
     * PLEASE NOTE: This method is for tests only
     *
     * @return
     */
    protected List<INDArray[]> getOutputs() {
        return outputs;
    }

    protected void setCounter(int value) {
        counter.set(value);
    }

    public void setPosition(int pos) {
        position.set(pos);
    }

    public int getCounter() {
        return counter.get();
    }



    public boolean isLocked() {
        boolean lck = !realLocker.readLock().tryLock();

        boolean result = lck || isLocked.get();

        if (!result)
            isReadLocked.set(true);

        return result;
    }


    @Override
    public INDArray[] getOutput() {
        // basically we should take care of splits here: each client should get its own part of output, wrt order number

        return outputs.get(position.get());
    }
}
